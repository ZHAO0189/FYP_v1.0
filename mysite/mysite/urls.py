from django.conf.urls import include, url
from django.contrib import admin
from django.conf import settings
from django.conf.urls.static import static



urlpatterns = [
    # Examples:
    # url(r'^$', 'mysite.views.home', name='home'),
    # url(r'^blog/', include('blog.urls')),

    url(r'^main/', 'testapp.views.main', name='main'),
    url(r'^crawling/', 'testapp.views.crawling', name='crawling'),
    url(r'^classification/', 'testapp.views.classification', name='classification'),
    url(r'^admin/', include(admin.site.urls)),

]

if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root = settings.STATIC_ROOT)
